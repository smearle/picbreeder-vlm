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

import client.*;

import javax.swing.*;

import java.awt.*;
import java.awt.event.*;
import java.io.IOException;

import client.evolution.*;
import client.renderers.SingleRenderer;
import client.server.ServerException;

public class ImagePanel extends JPanel implements MouseListener {
	private final int id;
	private Individual individual;
	
	private final int BORDER = 5;
	private final int WIDTH;
	private final int HEIGHT;
	
	private final Color SELECTED_COLOR;
	private final Color UNSELECTED_COLOR;
	
	public EvolutionController evolutionController;
	public int generationID;
	public int highlight_wait=0;
	
	public ImagePanel(int id) {
		super();
		
		WIDTH = ParameterTableInstance.get().getInteger("display", "width");
		HEIGHT = ParameterTableInstance.get().getInteger("display", "height");
		
		SELECTED_COLOR = Color.decode(ParameterTableInstance.get().getParameter("display", "selected color"));
		UNSELECTED_COLOR = Color.decode(ParameterTableInstance.get().getParameter("display", "unselected color"));

		//this.setToolTipText("Click if you like me, click again if you change your mind.");
		
		this.id = id;
		individual = null;
		
		this.addMouseListener(this);
		
		individual = null;
		this.setPreferredSize(new Dimension(WIDTH + BORDER * 2 + 1, HEIGHT + BORDER * 2 + 1));
	}
	
	public int getId() {
		return id;
	}
	
	public void setIndividual(Individual ind) {
		individual = ind;
		((test.ImagePhenotype)ind.getPhenotype()).setObserver(this);
		
	//	ind.getGenome().
		this.repaint();
	}
	
	public void paint(Graphics g) {
		super.paint(g);
		
		if(individual == null)
			return;

		Image image = (Image)individual.getPhenotype();
		if(individual.isSelected())
			g.setColor(SELECTED_COLOR);
		else
			g.setColor(UNSELECTED_COLOR);
		
		g.fillRect(0, 0, getWidth()-1, getHeight()-1);
		g.drawImage(image, BORDER, BORDER, this);
		
		if (false && individual.isSelected())
		{
			highlight_wait++;
			this.updateUI();
			if (highlight_wait>500)
			{
				try {
					client.utilities.XML.store(individual.getGenome(),"genomes/"+EvolutionController.starting_timestamp.getDateTime()+" "+generationID+"__genome_SELECTED.txt");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				evolutionController.spawn();
				highlight_wait = 0;
			}
			System.out.println(highlight_wait);
			//try {
				//this.wait(1000);
			//} catch (InterruptedException e) {
				// TODO Auto-generated catch block
			//	e.printStackTrace();
			//}

		}
	}
	
	public void mouseClicked(MouseEvent event) {
	}
	
	public void mouseEntered(MouseEvent event) {
	}
	
	public void mouseExited(MouseEvent event) {
	}
		
	public void mousePressed(MouseEvent event) {
		if(individual == null || highlight_wait>0)
			return;
		
		if(SwingUtilities.isLeftMouseButton(event)) {
			if(individual.isSelected())
				individual.deselect();
			else
			{
				individual.select();
				
				//Image image =  (Image)individual.getPhenotype())
				//for (int i=0; i<600; i++)
				//{
				//	int w = i; int h = i;
				//	Image scaledImage = image.getScaledInstance((int)w, (int)h, Image.SCALE_SMOOTH);
				//	drawImage(scaledImage, BORDER, BORDER, this);
				//}
				
				try {
					client.utilities.XML.store(individual.getGenome(),"genomes/"+EvolutionController.starting_timestamp.getDateTime()+" "+generationID+"__genome_SELECTED.txt");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				evolutionController.spawn();

			}
			repaint();
		}
	}
	
	public void mouseReleased(MouseEvent event) {
	}
	
}
