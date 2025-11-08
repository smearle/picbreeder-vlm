/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.gui;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.ByteArrayInputStream;

import javax.swing.*;

import client.MainComponentInstance;
import client.ParameterTableInstance;
import client.evolution.GeneticFactoryInstance;
import client.evolution.Genome;
import client.server.FatalException;
import client.server.ServerConnectionInstance;
import client.server.TimeoutException;
import client.utilities.ArgumentParser;

public class MainWindow extends JFrame implements SessionEnd, KeyListener {
	public MainWindow(String title) {
		this.setTitle(title);
		this.setUndecorated(true);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); //DO_NOTHING_ON_CLOSE);//
		this.addKeyListener(this);
	}
	
	
	public static void main(String [] args) {
		client.evolution.Series series = null;
		
		ArgumentParser parser = new ArgumentParser(args);

		client.server.ServerConnection server = null;
		
		try {
			if(parser.hasOption("--server")) {
				server = new client.server.SOAPServer();
				server.initialize(new test.TestInitialization());
			}
			else {
				server = new client.server.LocalServer();
				server.initialize(new client.server.FileInitialization());
			}
			
			SessionHandler.beginSession(server);
		}
		catch(SessionAlreadyExists e) {
			// only happens in the applet
			e.printStackTrace();
			return;
		}
		catch(TimeoutException e) {
			e.printStackTrace();
			return;
		}
		catch(Exception e) {
			e.printStackTrace();
		}

		client.ImageDatabaseInstance.set(new FileSystemImageDatabase());
		
		if(server.hasSeries()) {
			series = GeneticFactoryInstance.get().createInvalidSeries();
			
			java.io.InputStream xml = server.getSeries();
			client.utilities.XML.load(series, xml);
		}
		else if(server.hasGenome()) {
			Genome genome = GeneticFactoryInstance.get().createInvalidGenome();
			
			java.io.InputStream xml = server.getGenome();
			
			client.utilities.XML.load(genome, xml);
			series = GeneticFactoryInstance.get().createBranchSeries(genome);
		}
		else
			series = GeneticFactoryInstance.get().createRootSeries();

		server.clientStarted();
		SeriesInstance.set(series);
		
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				MainWindow window = new MainWindow(ParameterTableInstance.get().getParameter("meta", "project"));
				MainComponentInstance.set(window);

				
				window.add(new MainPanel());
				window.pack();
				window.setVisible(true);
				
				
			//	window.setSize( ParameterTableInstance.get().getInteger("display", "window_width") , ParameterTableInstance.get().getInteger("display", "window_height"));

				window.setResizable(false);
			}
		});
	}
	
	public void quit() {
		SessionHandler.endSession();
		System.exit(0);
	}
	
	public void publish() {
		SessionHandler.endSession();
		System.exit(0);
	}
	
	public void register() {
		SessionHandler.endSession();
		System.exit(0);
	}
	
	public void authenticate(String u, char []p) {
	}


	@Override
	public void keyPressed(KeyEvent e) {
		// TODO Auto-generated method stub 
		//System.out.println("KEY PRESSED "+e.isControlDown()+" "+e.getKeyCode()+" "+e.isShiftDown());
		if (e.isControlDown() && e.isShiftDown() && e.getKeyCode()==88) {
	     System.exit(0);
		}
	}


	@Override
	public void keyReleased(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void keyTyped(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}
}
