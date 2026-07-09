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
import client.evolution.*;

public class ImagePanel extends JPanel implements MouseListener {
	private final int id;
	private Individual individual;
	
	private final int BORDER = 5;
	private final int WIDTH;
	private final int HEIGHT;
	
	private final Color SELECTED_COLOR;
	private final Color UNSELECTED_COLOR;
	
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
		((test.ColorAndGreyIndividual)ind).setObserver(this);
		this.repaint();
	}
	
	public void paint(Graphics g) {
		super.paint(g);
		
		if(individual == null)
			return;

		Image image = (Image)individual.getPhenotype(individual.getGenome().getDominantPhenotype());
		if(individual.isSelected())
			g.setColor(SELECTED_COLOR);
		else
			g.setColor(UNSELECTED_COLOR);
		
		g.fillRect(0, 0, getWidth()-1, getHeight()-1);
		g.drawImage(image, BORDER, BORDER, this);
	}
	
	public void mouseClicked(MouseEvent event) {
	}
	
	public void mouseEntered(MouseEvent event) {
	}
	
	public void mouseExited(MouseEvent event) {
	}
	
	public void mousePressed(MouseEvent event) {
		if(individual == null)
			return;
		
		if(SwingUtilities.isLeftMouseButton(event)) {
			if(individual.isSelected())
				individual.deselect();
			else
				individual.select();
			repaint();
		}
	}
	
	public void mouseReleased(MouseEvent event) {
	}
	
}
